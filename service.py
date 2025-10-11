"""
service.py
Author: Matt Lindborg
Course: MS548 - Advanced Programming Concepts and AI
Assignment: Week 5
Date: 10/11/2025

Purpose:
This file implements the "business logic" for Learnflow Base.
The GUI (ui.py) delegates actions to this service layer.
Key responsibilities:
    - Add new learning entries (Goals, Skills, Sessions, Notes).
    - Store entries as LearningLog objects (defined in domain.py).
    - Provide summaries and history views for the GUI.
    - Clear/reset all entries.
This keeps the GUI and data model decoupled, enabling future
expansion (OOP classes, logfile persistence, AI integration).
"""

# --- Imports ---
from typing import Optional, Dict                          # type hinting for clarity
from copy import deepcopy                                  # for safe state snapshot
from domain import EntryType, LearnflowState, LearningLog  # import domain model classes
from textblob import TextBlob                              # import for sentiment analysis
from llama_cpp import Llama                                # import for ai llm library
import threading                                           # import for multi threading

# --- Individual Service Classes for specific applications ---
class TTSService:
    """
    Text-to-speech. Outs cleanly if pyttsx3 is not installed.
    Keeps UI separate from the business logic.
    """
    
    def __init__(self, enabled=True):
        self.enabled = enabled

    def set_enabled(self, value: bool):
        """Toggle TTS on/off from the checkbox in UI."""
        self.enabled = value

    def speak(self, text: str):
        """
        Speak text asynchronously using pyttsx3.
        A new engine instance is created for each call to avoid
        the Windows 'silent after first run' bug when threaded.
        """
        if not (self.enabled and text):
            return

        def run_tts():
            try:
                import pyttsx3
                engine = pyttsx3.init()
                engine.setProperty("rate", 180)
                for v in engine.getProperty("voices"):
                    if "female" in v.name.lower() or "zira" in v.name.lower():
                        engine.setProperty("voice", v.id)
                        break
                engine.say(text)
                engine.runAndWait()
                engine.stop()
            except Exception as e:
                print(f"TTS error: {e}")

        # launch speech in a daemon thread so it doesn't block the GUI
        threading.Thread(target=run_tts, daemon=True).start()

class LearnflowService:
    """
    The LearnflowService class for all non-UI functionality.
    It operates on a LearnflowState object, which stores user data.
    """

    def __init__(self, state: Optional[LearnflowState] = None):
        """
        Constructor initializes service with an existing state,
        or creates a new empty LearnflowState if none provided.
        """
        self._state = state or LearnflowState()
        self.responses = LlamaEngine() # full ai replies
        prompt_text = self.get_prompt() # pull the prompt text
        if prompt_text:
            self.responses.system_prompt = prompt_text
        self.tts = TTSService()  # audio on by default

    # ------------------- COMMANDS (Mutate State) -------------------

    def set_entry(self, entry_type: EntryType, text: str) -> None:
        """
        Add a new entry to the state.
        Input:
            entry_type - The category of entry (Goal, Skill, Session, Notes).
            text       - The user-provided content string.
        Behavior:
            - Creates a new LearningLog object.
            - Appends it to the list for the given entry_type.
            - Calls write_log() stub for future logfile use.
        """
        # sanitize text
        clean_text = (text or "").strip()

        # delegated mood evaluation to a helper method
        mood = self.analyze_mood(clean_text)

        # create new log record object with mood
        record = LearningLog(entry_type, clean_text, mood=mood)

        # append to the appropriate list in state
        self._state.entries[entry_type].append(record)

        # placeholder hook for logfile writing
        self.write_log(record)

    def clear(self) -> None:
        """
        Reset the entire state back to empty lists.
        Useful for starting over without restarting the program.
        """
        for k in self._state.entries:
            self._state.entries[k] = []
            
    def set_llm(self, model_path: str):
        """
        Dynamically reload the Llama model with a new GGUF file.
        """
        try:
            print(f"Loading new model: {model_path}")
            self.responses = LlamaEngine(model_path=model_path)
            return f"Model loaded: {model_path}"
        except Exception as e:
            return f"Error loading model: {e}"
        
    def get_current_model(self) -> str:
        """
        Return the current LLM model file name (for menu display).
        """
        try:
            import os
            return os.path.basename(self.responses.llm.model_path)
        except Exception:
            return "Unknown Model"

    # ------------------- QUERIES (Read State) -------------------

    def get_entry(self, entry_type: EntryType) -> str:
        """
        Retrieve the most recent entry for a given type.
        Returns:
            - The latest entry text if available.
            - Empty string if no entries exist for this type.
        """
        if self._state.entries[entry_type]:
            return self._state.entries[entry_type][-1].text
        return ""

    def summary(self) -> Dict[str, str]:
        """
        Build a dictionary summary of the most recent entries by type.
        Returns:
            { "Goal": "Finish Week 1", "Notes": "Felt motivated", ... }
        Each value comes from the LearningLog.summary() method.
        """
        result = {}
        for e, records in self._state.entries.items():
            if records:  # only include if there is at least one record
                result[e.value] = records[-1].summary()
        return result

    def snapshot(self) -> LearnflowState:
        """
        Return a deep copy of the current LearnflowState.
        This allows the GUI to display history safely without
        risking accidental modification of the underlying data.
        """
        return deepcopy(self._state)

    # ------------------- HELPER FUNCTIONS -------------------

    def speak_if_enabled(self, text: str) -> None:
        """
        Speak a line through TTS if the user has audio enabled.
        """
        self.tts.speak(text)

    def analyze_mood(self, text: str) -> str:
        """
        Run sentiment analysis on note text using TextBlob.
        Returns one of: "motivated", "stuck", or "neutral".
        Polarity amounts chosen for simplicity.
        """
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        if polarity > 0.3:      # positive sentiment
            return "motivated"
        elif polarity < -0.3:   # negative sentiment
            return "stuck"
        else:                   # neutral sentiment
            return "neutral"

    def write_log(self, record: "LearningLog"):
        """
        Append a log entry to a persistent text file (learnflow.log).
        Subclass-aware:
        - GoalLog → includes Status
        - ReflectionLog → includes Mood
        - LearningLog → base summary
        """
        from domain import GoalLog, ReflectionLog

        log_file = "learnflow.log"

        # base summary always includes entry type and text
        line = f"[{record.timestamp}] {record.entry_type.value}: {record.text}"

        # add subclass-specific info
        if isinstance(record, GoalLog):
            line += f" (Status: {record.status})"
        elif isinstance(record, ReflectionLog):
            if record.mood:
                line += f" (Mood: {record.mood})"
        elif record.mood:  # base LearningLog may still carry a mood
            line += f" (Mood: {record.mood})"

        # write line to logfile
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(line + "\n")
            
    # ------------------- SESSION LOGGING -------------------

    def save_session_log(self, user_input: str, ai_output: str):
        """
        Save each user/AI message pair to a session_log.json file.
        The log is later used to continue conversation context.
        """
        import json, os
        log_file = "session_log.json"
        entry = {"user": user_input, "ai": ai_output}

        # load existing log or start new one
        if os.path.exists(log_file):
            try:
                with open(log_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError):
                data = []
        else:
            data = []

        data.append(entry)

        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

    def load_session_log(self) -> list:
        """
        Load the previous session log so the LLM can retain context.
        Returns a list of prior exchanges.
        """
        import json, os
        log_file = "session_log.json"
        if not os.path.exists(log_file):
            return []

        with open(log_file, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return []
            
    # --- Session Chat File Logging ---
    def update_chat_log(self, full_text: str, append: bool = False):
        """
        Write or append the complete AI chat text to a persistent log file.
        Called whenever the AI output box is updated.
        """
        log_path = "chat_history.txt"
        mode = "a" if append else "w"
        try:
            with open(log_path, mode, encoding="utf-8") as f:
                f.write(full_text if append else full_text.rstrip() + "\n")
        except Exception as e:
            print(f"[WARN] Failed to write chat log: {e}")
                
    # ------------------- PROMPT MANAGEMENT -------------------

    def set_prompt(self, text: str):
        """
        Save a custom system prompt to prompt.txt.
        """
        with open("prompt.txt", "w", encoding="utf-8") as f:
            f.write(text.strip())

        if hasattr(self, "responses"):
            self.responses.system_prompt = text.strip()

    def get_prompt(self) -> str:
        """
        Load the current system prompt from file or return default.
        """
        import os
        if os.path.exists("prompt.txt"):
            with open("prompt.txt", "r", encoding="utf-8") as f:
                return f.read().strip()
        return getattr(self.responses, "system_prompt", "")
    
    def reset_context(self):
        """
        Clear conversation log and reload the LLM context.
        """
        import os
        if os.path.exists("session_log.json"):
            os.remove("session_log.json")
        self.responses.reset_context()
        return "Session has been refreshed."

class LlamaEngine:
    """
    Wrapper around llama-cpp to generate responses from a local GGUF model.
    Includes support for persistent prompt and session history.
    """
    def __init__(self, model_path="llm/Hermes-3-Llama-3.1-8B.Q5_K_M.gguf"):
        self.model_path = model_path
        self.llm = Llama(model_path=model_path, n_ctx=2048, n_threads=8, verbose=True)
        self.system_prompt = "You are a friendly learning advisor who motivates students kindly."
        self.context = []

    def reply(self, user_text: str) -> str:
        """
        Send user input to the LLM and return response, maintaining context.
        """
        full_prompt = self.system_prompt + "\n"
        for msg in self.context[-5:]:  # keep last 5 for brevity
            full_prompt += f"User: {msg['user']}\nAssistant: {msg['ai']}\n"

        full_prompt += f"User: {user_text}\nAssistant:"
        response = self.llm(full_prompt, max_tokens=512, stop=["User:"], echo=False)
        reply = response["choices"][0]["text"].strip()

        # save context locally
        self.context.append({"user": user_text, "ai": reply})
        return reply

    def reset_context(self):
        """
        Clears in-memory conversation context.
        """
        self.context = []
    
    