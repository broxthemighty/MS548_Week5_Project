"""
ui.py
Author: Matt Lindborg
Course: MS548 - Advanced Programming Concepts and AI
Assignment: Week 5
Date: 10/11/2025

Purpose:
This file defines the Tkinter-based graphical user interface (GUI)
for the Learnflow Base application. The GUI layer is responsible for:
    - Displaying buttons, menus, and text areas to the user.
    - Collecting input via popup dialogs.
    - Rendering summaries and history of entries.
    - Delegating business logic to the LearnflowService (service.py).

This file does NOT contain data storage logic. Instead:
    - It calls service methods to set/get data.
    - It uses the domain model (LearningLog, EntryType) indirectly.
    - It is designed to be event-driven (each button triggers a method).
"""

# --- Imports ---
# standard library
import json                            # for save/load functionality
import csv                             # excel file output
import tkinter as tk
from tkinter import filedialog         # standard Tkinter dialogs

# local
from service import LearnflowService   # service layer abstraction
from domain import EntryType           # domain layer EntryType


class AutoScrollbar(tk.Scrollbar):
    """
    A scrollbar that hides itself when not needed.
    Works only with grid geometry.
    """
    def set(self, lo, hi):
        # hide scrollbar if the text fits entirely in the view
        if float(lo) <= 0.0 and float(hi) >= 1.0:
            self.grid_remove()
        else:
            self.grid()
        super().set(lo, hi)

    def pack(self, **kw):
        raise tk.TclError("Use grid instead of pack with AutoScrollbar")

    def place(self, **kw):
        raise tk.TclError("Use grid instead of place with AutoScrollbar")

class App:
    """
    The App class defines the GUI layout and event handlers.
    It is initialized with a root Tkinter window and a LearnflowService.
    """

    def __init__(self, root: tk.Tk, service: LearnflowService):
        """
        Constructor initializes the window, builds the layout,
        and renders the initial summary display.
        """
        # hold references to the Tk root and the business service
        self.root = root
        self.service = service

        # configure window title and disable resizing
        self.root.title("Learnflow Base")
        self.root.resizable(False, False)
        self.root.geometry("625x840") # hard coded to not waste space

        # set base background and foreground
        self.root.option_add("*Background", "#2b2b2b")      # dark gray background
        self.root.option_add("*Foreground", "#ffffff")      # white text

        # button colors
        self.root.option_add("*Button.Background", "#444444")
        self.root.option_add("*Button.Foreground", "#ffffff")
        self.root.option_add("*Button.ActiveBackground", "#666666")
        self.root.option_add("*Button.ActiveForeground", "#ffffff")

        # entry box colors
        self.root.option_add("*Entry.Background", "#1e1e1e")
        self.root.option_add("*Entry.Foreground", "#dcdcdc")
        self.root.option_add("*Entry.InsertBackground", "#ffffff")

        # text box colors
        self.root.option_add("*Text.Background", "#1e1e1e")
        self.root.option_add("*Text.Foreground", "#dcdcdc")

        # menu colors
        self.root.option_add("*Menu.Background", "#2b2b2b")
        self.root.option_add("*Menu.Foreground", "#ffffff")
        self.root.option_add("*Menu.ActiveBackground", "#444444")
        self.root.option_add("*Menu.ActiveForeground", "#ffffff")

        # global font setting
        default_font = ("Segoe UI", 10)

        # --- Main container frame ---
        main_frame = tk.Frame(root, padx=10, pady=10)
        main_frame.grid(row=0, column=0, sticky="nw")
        #main_frame.columnconfigure(0, weight=1)

        # --- Top row: welcome label, clear button, drop-down menu ---
        top_frame = tk.Frame(main_frame)
        top_frame.grid(row=0, column=0, sticky="ew")
        top_frame.columnconfigure(0, weight=1)

        # main title label
        self.display_label = tk.Label(
            top_frame,
            text="Welcome to Learnflow\nMy name is Verita",
            font=("Georgia", 14),
            pady=2,
            justify="left",
        )
        self.display_label.grid(row=0, column=0, sticky="w")

        # frame to hold summary text and scrollbar (so scrollbar can hide itself if not needed)
        summary_container = tk.Frame(top_frame)
        summary_container.grid(row=0, column=2, padx=(5, 5), sticky="n")

        # auto-hiding vertical scrollbar for the summary box
        summary_scroll = AutoScrollbar(summary_container, orient="vertical")
        summary_scroll.grid(row=0, column=1, sticky="ns")

        # summary text box
        self.summary_box = tk.Text(
            summary_container,
            height=4,
            width=40,
            wrap="word",
            state="disabled",
            font=default_font,
            yscrollcommand=summary_scroll.set
        )
        self.summary_box.grid(row=0, column=0, sticky="nsew")

        # link scrollbar back to summary box
        summary_scroll.config(command=self.summary_box.yview)

        # allow the text box to expand properly inside the container
        summary_container.rowconfigure(0, weight=1)
        summary_container.columnconfigure(0, weight=1)

        # clear button
        self.clear_button = tk.Button(
            top_frame, text="Clear", width=7, command=self.clear_entries
        )
        self.clear_button.grid(row=0, column=1, sticky="w", padx=(5, 2))

        # attach a menubar
        self.build_menu()

        # --- Middle row: buttons for Goal/Skill/Session/Notes ---
        middle_frame = tk.Frame(main_frame)
        middle_frame.grid(row=1, column=0, sticky="nsew", pady=5)

        # image on the left
        try:
            self.image = tk.PhotoImage(file="images\\image2_50pc.png")
            self.image_label = tk.Label(middle_frame, image=self.image, width=512, height=512)
            self.image_label.pack(side="left", padx=(0, 10))
        except Exception:
            # fail gracefully if image not found
            pass
        
        # image on the left (animated) future integration, maybe use textblob for emotions on specific images
        try:
            import itertools
            from PIL import Image, ImageTk
            self.image_frames = [ImageTk.PhotoImage(Image.open(f"images/frame_{i}.png").resize((512, 512))) for i in range(1, 5)]
            self.image_label = tk.Label(middle_frame, image=self.image_frames[0])
            self.image_label.pack(side="left", padx=(0, 10))

            def animate(counter=itertools.cycle(range(len(self.image_frames)))):
                self.image_label.configure(image=self.image_frames[next(counter)])
                self.root.after(200, lambda: animate(counter))
            animate()
        except Exception as e:
            print(f"Animation load failed: {e}")

        # right frame with stacked buttons and log box
        right_frame = tk.Frame(middle_frame)
        right_frame.pack(side="left", anchor="n")

        # button frame
        buttons_frame = tk.Frame(right_frame)
        buttons_frame.pack(side="left", anchor="n", padx=(0, 5))

        # create one button per EntryType
        for et in (EntryType.Goal, EntryType.Skill, EntryType.Session, EntryType.Notes):
            tk.Button(
                buttons_frame,
                text=et.value,
                width=10,
                command=lambda t=et: self.on_add_or_edit_entry(t),
            ).pack(pady=2, anchor="w")

        # tts audio checkbox
        self.audio_var = tk.BooleanVar(value=True)
        audio_chk = tk.Checkbutton(
            buttons_frame,
            text="Audio",
            variable=self.audio_var,
            command=lambda: self.service.tts.set_enabled(self.audio_var.get())
        )
        audio_chk.pack(pady=(20, 5), anchor="w")

        # bottom row: ai input and responses output box ---
        ai_frame = tk.Frame(main_frame)
        ai_frame.grid(row=3, column=0, sticky="ew", pady=(0, 5), padx=(0, 5))

        # input field for user prompt to AI
        self.ai_entry = tk.Entry(
            ai_frame, 
            width=60, 
            font=default_font
            )
        self.ai_entry.insert(0, "Type your question for Verita...")

        # remove placeholder text when clicking into the box
        self.ai_entry.bind("<FocusIn>", self.clear_placeholder)

        # pressing Enter should submit text
        self.ai_entry.bind("<Return>", self.submit_ai_text)

        # detect typing so we can shift focus logic if needed
        self.ai_entry.bind("<KeyRelease>", self.focus_send_button)

        # location in the frame
        self.ai_entry.pack(side="left", fill="x", expand=True, padx=(0, 5))

        # send button
        self.ai_send_button = tk.Button(
            ai_frame,
            text="Send",
            command=lambda: self.display_ai_response(self.ai_entry.get())
        )
        self.ai_send_button.pack(side="right")
        
        # speech-to-text button (microphone)
        mic_button = tk.Button(
            ai_frame,
            text="🎙 Speak",
            command=self.speech_to_text
        )
        mic_button.pack(side="right", padx=(5, 5))

        # ai output frame
        ai_output_frame = tk.Frame(main_frame)
        ai_output_frame.grid(row=4, column=0, sticky="w", pady=(2, 0))

        # container frame for AI output text and scrollbar
        ai_output_container = tk.Frame(ai_output_frame)
        ai_output_container.pack(fill="both", expand=True)

        # always-visible vertical scrollbar
        ai_scroll = tk.Scrollbar(ai_output_container, orient="vertical")
        ai_scroll.pack(side="right", fill="y")

        # AI output text box with scrollbar attached
        self.ai_output_box = tk.Text(
            ai_output_container,
            width=83,
            height=10,
            wrap="word",
            state="normal",
            font=default_font,
            yscrollcommand=ai_scroll.set
        )
        self.ai_output_box.pack(side="left", fill="both", expand=True)
        ai_scroll.config(command=self.ai_output_box.yview)
        
        # timer label under AI output box
        self.response_time_label = tk.Label(
            ai_output_frame,
            text="Response time: —",
            anchor="w",  # left align
            font=("Segoe UI", 9, "italic")
        )
        self.response_time_label.pack(fill="x", padx=5, pady=(0, 5), anchor="w")

        # insert placeholder text at start
        self.ai_output_box.insert(tk.END, "Your learning journey with Verita begins here...\n")
        self.ai_output_box.config(state="disabled")

        # link scrollbar back to AI output box
        ai_scroll.config(command=self.ai_output_box.yview)

        # allow expansion inside container
        ai_output_container.rowconfigure(0, weight=1)
        ai_output_container.columnconfigure(0, weight=1)

        # initial render from service
        self.render_summary()

    # ------------------- HELPERS -------------------
    def custom_input_popup(
    self,
    title: str,
    prompt: str = "",
    ok_text: str = "OK",
    show_cancel: bool = False,
    multiline: bool = False
) -> str | None:
        """
        Custom popup dialog for text input.
        - title: popup window title
        - prompt: text shown above or pre-filled inside the box
        - ok_text: label for the confirmation button ("OK" or "Save")
        - show_cancel: whether to display a Cancel button
        - multiline: whether to use a Text box (multi-line) or Entry (single line)
        Returns the entered string or None if canceled/closed.
        """
        popup = tk.Toplevel(self.root)
        popup.title(title)
        width, height = (400, 250) if multiline else (300, 150)
        self.center_popup(popup, width, height)

        # label for instruction or context
        if not multiline and prompt:
            tk.Label(
                popup,
                text=prompt,
                font=("Segoe UI", 9),
                wraplength=width - 40,
                justify="left"
            ).pack(pady=(10, 5), padx=10)

        result = {"value": None}

        # multi-line text or single-line entry
        if multiline:
            text_widget = tk.Text(
                popup,
                width=45,
                height=6,
                wrap="word",
                font=("Segoe UI", 10),
                bg="#1e1e1e",
                fg="#dcdcdc",
                insertbackground="#ffffff"
            )
            text_widget.insert("1.0", prompt or "")
            text_widget.pack(padx=10, pady=(10, 10), fill="both", expand=True)
            text_widget.focus_set()
        else:
            text_widget = tk.Entry(popup, width=40)
            text_widget.insert(0, prompt or "")
            text_widget.pack(pady=10)
            text_widget.focus_set()

        # button frame for OK / Save / Cancel
        btn_frame = tk.Frame(popup, bg="#2b2b2b")
        btn_frame.pack(pady=(5, 10))

        def on_ok(event=None):
            val = text_widget.get("1.0", "end-1c") if multiline else text_widget.get()
            result["value"] = val.strip()
            popup.destroy()

        def on_cancel():
            result["value"] = None
            popup.destroy()

        ok_btn = tk.Button(
            btn_frame,
            text=ok_text,
            bg="#444444",
            fg="#ffffff",
            activebackground="#666666",
            activeforeground="#ffffff",
            command=on_ok
        )
        ok_btn.pack(side="left", padx=5)

        if show_cancel:
            cancel_btn = tk.Button(
                btn_frame,
                text="Cancel",
                bg="#444444",
                fg="#ffffff",
                activebackground="#666666",
                activeforeground="#ffffff",
                command=on_cancel
            )
            cancel_btn.pack(side="left", padx=5)

        popup.bind("<Return>", on_ok)
        self.root.wait_window(popup)
        return result["value"]
    
    def custom_message_popup(self, title: str, message: str, msg_type: str = "info"):
        """
        Custom message popup to replace default messagebox dialogs.
        - title: window title text
        - message: main message content
        - msg_type: "info", "error", "warning" (changes color scheme)
        Enhanced version: wraps text and auto-sizes for readability.
        """

        popup = tk.Toplevel(self.root)
        popup.title(title)

        # determine size based on message length
        base_width = 350
        base_height = 180
        if len(message) > 150:
            base_height = 250
        if len(message) > 300:
            base_height = 320

        self.center_popup(popup, base_width, base_height)

        # choose colors based on message type
        if msg_type == "error":
            bg_color = "#5c2b2b"
            fg_color = "#ffcccc"
        elif msg_type == "warning":
            bg_color = "#5c5c2b"
            fg_color = "#ffffcc"
        else:  # info
            bg_color = "#2b2b2b"
            fg_color = "#ffffff"

        popup.configure(bg=bg_color)

        # scrollable frame for long messages
        import tkinter.scrolledtext as st
        text_area = st.ScrolledText(
            popup,
            wrap="word",
            height=6,
            font=("Segoe UI", 10),
            bg=bg_color,
            fg=fg_color,
            relief="flat",
            borderwidth=0
        )
        text_area.insert("1.0", message)
        text_area.config(state="disabled")
        text_area.pack(pady=(15, 10), padx=10, fill="both", expand=True)

        # OK button to close popup
        ok_button = tk.Button(
            popup,
            text="OK",
            bg="#444444",
            fg="#ffffff",
            activebackground="#666666",
            activeforeground="#ffffff",
            command=popup.destroy
        )
        ok_button.pack(pady=(0, 10))
        ok_button.focus_set()
        popup.bind("<Return>", lambda event=None: popup.destroy())
        return popup

    def render_summary(self) -> None:
        """
        Render the latest entries (summary) in the bottom output box.
        """
        summary = self.service.summary()

        # update summary box
        self.summary_box.config(state="normal")
        self.summary_box.delete("1.0", tk.END)
        for val in summary.values():
            self.summary_box.insert(tk.END, f"{val}\n")
        self.summary_box.config(state="disabled")

    def clear_placeholder(self, event):
        """
        Remove placeholder text when user clicks into the entry box.
        """
        if self.ai_entry.get().strip() == "Type your question for Verita...":
            self.ai_entry.delete(0, tk.END)
            self.ai_entry.unbind("<FocusIn>")

    def focus_send_button(self, event):
        """
        If the user has started typing real text,
        keep focus in the entry box until Enter is pressed.
        """
        current_text = self.ai_entry.get().strip()
        if current_text and current_text != "Type your question for Verita...":
            # keep focus in the entry so user can continue typing
            self.ai_entry.focus_set()

    def _handle_ai_input(self, user_input: str):
        """
        Core logic for processing AI input and updating UI.
        Handles AI input: shows 'processing...', then replaces it with the LLM's reply.
        """
        user_input = user_input.strip()
        if not user_input or user_input == "Type your question for Verita...":
            return

        self.ai_output_box.config(state="normal")
        self.ai_output_box.insert(tk.END, f"You: {user_input}\n")

        # record index BEFORE inserting placeholder
        insert_start = self.ai_output_box.index(tk.END)
        self.ai_output_box.insert(tk.END, "Verita: ...processing...\n\n")
        self.ai_output_box.see(tk.END)
        self.ai_output_box.config(state="disabled")

        import threading
        import time

        def run_llm():
            try:
                # start timer
                start_time = time.time()
                self.root.after(0, lambda: self.response_time_label.config(text="Processing...", fg="#888888"))

                # run model
                reply = self.service.responses.reply(user_input)

                # end timer
                end_time = time.time()
                elapsed = end_time - start_time

                # save to log file
                self.service.save_session_log(user_input, reply)

                def update_text():
                    # re-enable text box for modification
                    self.ai_output_box.config(state="normal")

                    # remove the placeholder ("processing...") line only
                    self.ai_output_box.delete(f"{insert_start} linestart", f"{insert_start} lineend+1c")

                    # insert the actual response at that same spot
                    self.ai_output_box.insert(insert_start, f"Verita: {reply}\n\n")
                    self.ai_output_box.see(tk.END)
                    self.ai_output_box.config(state="disabled")

                    # update response time label (show in color)
                    if elapsed < 2:
                        color = "green"
                    elif elapsed < 5:
                        color = "orange"
                    else:
                        color = "red"

                    self.response_time_label.config(
                        text=f"Response time: {elapsed:.2f} seconds",
                        fg=color
                    )

                    # write the current conversation to chat_history.txt
                    full_text = self.ai_output_box.get("1.0", "end-1c")
                    self.service.update_chat_log(full_text)
                    
                    # speak the response via TTS
                    self.service.speak_if_enabled(reply)

                # schedule UI update on main thread
                self.root.after(0, update_text)

            except Exception as e:
                self.root.after(0, lambda: self.custom_message_popup("Error", f"AI Error: {e}", msg_type="error"))

        # start the background thread here
        threading.Thread(target=run_llm, daemon=True).start()

        # clear entry box after starting the thread
        self.ai_entry.delete(0, tk.END)
        self.ai_entry.focus_set()
        def _insert_ai_reply(self, reply: str):
            self.ai_output_box.config(state="normal")
            self.ai_output_box.insert(tk.END, f"AI: {reply}\n\n")
            self.ai_output_box.see(tk.END)
            self.ai_output_box.config(state="disabled")
            self.service.speak_if_enabled(reply)
        
    def speech_to_text(self):
        """
        Convert speech input to text and insert it into the entry box.
        """
        import threading

        def record_audio():
            try:
                import speech_recognition as sr
                r = sr.Recognizer()

                # adjust sensitivity to ambient noise
                with sr.Microphone() as source:
                    popup = self.custom_message_popup("Listening", "Speak now…", msg_type="info")

                    # optional ambient calibration (increases reliability in noisy rooms)
                    r.adjust_for_ambient_noise(source, duration=0.8)

                    # increase energy threshold for clarity and allow longer phrases
                    r.energy_threshold = 300
                    r.pause_threshold = 1.2  # seconds of silence before stopping
                    r.phrase_threshold = 0.3
                    r.non_speaking_duration = 0.5

                    # listen for speech up to 20 seconds total
                    audio = r.listen(source, timeout=8, phrase_time_limit=20)

                # recognize speech with Google Speech Recognition
                text = r.recognize_google(audio)
                
                # close the listening popup
                if popup and popup.winfo_exists():
                    popup.destroy()

                # update GUI input box
                self.ai_entry.delete(0, tk.END)
                self.ai_entry.insert(0, text)
                self._handle_ai_input(text)

            except sr.WaitTimeoutError:
                self.custom_message_popup("Timeout", "No speech detected. Please try again.", msg_type="warning")
            except sr.UnknownValueError:
                self.custom_message_popup("Error", "Speech was unclear. Please try again.", msg_type="error")
            except sr.RequestError as e:
                self.custom_message_popup("Error", f"Speech API unavailable: {e}", msg_type="error")
            except Exception as e:
                self.custom_message_popup(
                    "Speech Error",
                    f"Speech recognition failed: {e}\nTry reinstalling PyAudio or checking your microphone.",
                    msg_type="error"
                )

        threading.Thread(target=record_audio, daemon=True).start()

    def submit_ai_text(self, event=None):
        """
        Triggered when pressing Enter.
        """
        self._handle_ai_input(self.ai_entry.get())

    def display_ai_response(self, user_input: str):
        """
        Triggered when clicking Send.
        """
        self._handle_ai_input(user_input)

    # ------------------- EVENT HANDLERS -------------------
    def on_add_or_edit_entry(self, entry_type: EntryType):
        """
        Event handler for Goal/Skill/Session/Notes buttons.
        Steps:
            - Open custom popup for user input.
            - If Goal then create a GoalLog (with status).
            - If Notes then create a ReflectionLog (with mood analysis).
            - Otherwise then use normal LearningLog via service.
            - Re-render summary output.
        """
        text = self.custom_input_popup("Input", f"Enter your {entry_type.value}:")
        if not text:
            return  # user canceled

        # sentiment for all entries
        self.service.set_entry(entry_type, text)

        self.render_summary()

    def clear_entries(self) -> None:
        """
        Clear all entries from the service and refresh display.
        """
        self.service.clear()
        self.render_summary()
        self.custom_message_popup("Cleared", "All entries have been cleared.", msg_type="info")

    # ------------------- MENU & FILE OPS -------------------
    def build_menu(self):
        """
        Build the top menubar with File menu options.
        """
        menubar = tk.Menu(self.root)

        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Save", command=self.save_entries) # save data entries in json formatted file
        file_menu.add_command(label="Load", command=self.load_entries) # load data entries in json formatted file
        file_menu.add_command(label="Export CSV", command=self.export_csv)  # export history to Excel-readable format
        file_menu.add_command(label="View History", command=self.show_history)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=file_menu)
        
        self.root.config(menu=menubar)
        
        # add llm model selection to menu bar
        llm_menu = tk.Menu(menubar, tearoff=0)
        
        # display current model name (disabled so not clickable)
        current_model = self.service.get_current_model() if hasattr(self.service, "get_current_model") else "Unknown"
        llm_menu.add_command(label=f"Current Model: {current_model}", state="disabled")

        # separator then load option
        llm_menu.add_separator()
        llm_menu.add_command(label="Select Model", command=self.load_llm) # change the llm in use
        llm_menu.add_command(label="Change Avatar", command=self.change_avatar) # add Change Avatar option
        menubar.add_cascade(label="LLM", menu=llm_menu)
        
        # store a reference so we can update the label later when model changes
        self.llm_menu = llm_menu
        
        # prompt menu for viewing/editing/saving prompt
        prompt_menu = tk.Menu(menubar, tearoff=0)
        prompt_menu.add_command(label="View/Edit Prompt", command=self.edit_prompt)
        prompt_menu.add_command(label="Save Prompt", command=self.save_prompt)
        prompt_menu.add_command(label="Current Prompt", command=self.load_prompt)
        menubar.add_cascade(label="Prompt", menu=prompt_menu)

        # session controls
        session_menu = tk.Menu(menubar, tearoff=0)
        session_menu.add_command(label="Refresh Session", command=self.refresh_session)
        menubar.add_cascade(label="Session", menu=session_menu)

        # add Entries menu with entry-related actions
        entries_menu = tk.Menu(menubar, tearoff=0)
        entries_menu.add_command(label="Goal", command=lambda: self.on_add_or_edit_entry(EntryType.Goal))
        entries_menu.add_command(label="Skill", command=lambda: self.on_add_or_edit_entry(EntryType.Skill))
        entries_menu.add_command(label="Session", command=lambda: self.on_add_or_edit_entry(EntryType.Session))
        entries_menu.add_command(label="Notes", command=lambda: self.on_add_or_edit_entry(EntryType.Notes))
        entries_menu.add_separator()
        entries_menu.add_command(label="Clear", command=self.clear_entries)
        menubar.add_cascade(label="Entries", menu=entries_menu)

    def save_entries(self):
        """
        Save all current entries to a JSON file.
        Explicitly writes base attributes and subclass-specific ones.
        - LearningLog = entry_type, text, timestamp, mood
        - GoalLog = adds 'status'
        - ReflectionLog = keeps 'mood'
        """
        file_path = filedialog.asksaveasfilename(
            defaultextension=".json", filetypes=[("JSON files", "*.json")]
        )
        if not file_path:
            return

        history = self.service.snapshot().entries

        from domain import GoalLog, ReflectionLog

        export_dict = {}

        for et, logs in history.items():
            if logs:
                export_dict[et.value] = []
                for rec in logs:
                    # base record dictionary
                    record_dict = {
                        "entry_type": et.value,
                        "text": rec.text,
                        "timestamp": rec.timestamp,
                        "mood": getattr(rec, "mood", "")
                    }

                    # add subclass-specific attributes
                    if isinstance(rec, GoalLog):
                        record_dict["status"] = rec.status
                    elif isinstance(rec, ReflectionLog):
                        record_dict["mood"] = rec.mood  # mood is present

                    export_dict[et.value].append(record_dict)

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(export_dict, f, indent=4)

        self.custom_message_popup("Saved", f"Entries saved to {file_path}", msg_type="info")

    def load_entries(self):
        """
        Load entries from a JSON file.
        Reconstructs the correct class type:
        - GoalLog if 'status' field is present
        - ReflectionLog if entry_type == 'Notes'
        - LearningLog otherwise
        """
        file_path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
        if not file_path:
            return

        from domain import GoalLog, ReflectionLog, LearningLog, EntryType

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # reset current state before loading
            self.service._state.entries = {e: [] for e in EntryType}

            for etype_str, records in data.items():
                etype = EntryType(etype_str)
                for rec in records:
                    text = rec.get("text", "")
                    timestamp = rec.get("timestamp", "")
                    mood = rec.get("mood", "")

                    if "status" in rec:
                        # build GoalLog
                        entry = GoalLog(etype, text, timestamp=timestamp, mood=mood, status=rec["status"])
                    elif etype == EntryType.Notes:
                        # build ReflectionLog
                        entry = ReflectionLog(etype, text, timestamp=timestamp, mood=mood)
                    else:
                        # build base LearningLog
                        entry = LearningLog(etype, text, timestamp=timestamp, mood=mood)

                    self.service._state.entries[etype].append(entry)

            self.render_summary()
            self.custom_message_popup("Loaded", f"Entries loaded from {file_path}", msg_type="info")

        except Exception as e:
            self.custom_message_popup("Error", "Failed to load file!", msg_type="error")

    def show_history(self):
        """
        Display the AI chat history from chat_history.txt.
        Opens the saved transcript file (if it exists) in a scrollable popup.
        """
        import os

        log_file = "chat_history.txt"
        if not os.path.exists(log_file):
            self.custom_message_popup("Chat History", "No chat history found yet.")
            return

        popup = tk.Toplevel(self.root)
        popup.title("Chat History")
        self.center_popup(popup, 650, 450)

        scrollbar = tk.Scrollbar(popup)
        scrollbar.pack(side="right", fill="y")

        text_area = tk.Text(popup, wrap="word", yscrollcommand=scrollbar.set)
        text_area.pack(fill="both", expand=True)
        scrollbar.config(command=text_area.yview)

        try:
            with open(log_file, "r", encoding="utf-8") as f:
                content = f.read()
            text_area.insert(tk.END, content)
        except Exception as e:
            text_area.insert(tk.END, f"Failed to read log: {e}")

        text_area.config(state="disabled")


    def load_llm(self):
        """
        Allows the user to select and load a different local LLM (GGUF file).
        """
        file_path = filedialog.askopenfilename(
            title="Select GGUF Model File",
            filetypes=[("GGUF files", "*.gguf"), ("All files", "*.*")]
        )
        if not file_path:
            return
        result = self.service.set_llm(file_path)
        self.custom_message_popup("LLM Model", result, msg_type="info")
        
        # update menu label to show new model name
        current_model = self.service.get_current_model()
        self.llm_menu.entryconfig(0, label=f"Current Model: {current_model}")

    def center_popup(self, popup, width, height):
        """
        Center any popup relative to the main app window.
        """
        self.root.update_idletasks()

        # grab the info for the root frame
        main_x = self.root.winfo_x()
        main_y = self.root.winfo_y()
        main_w = self.root.winfo_width()
        main_h = self.root.winfo_height()

        # calculate horizontal offset: start from the root x position,
        # then add half the root width, and subtract half the popup width
        pos_x = main_x + (main_w // 2) - (width // 2)

        # calculate vertical offset: start from the root y position,
        # then add half the root height, and subtract half the popup height
        pos_y = main_y + (main_h // 2) - (height // 2)

        # apply the new location for the popup by changing it's geometry
        popup.geometry(f"{width}x{height}+{pos_x}+{pos_y}")
        
    def change_avatar(self):
        """
        Allow user to choose a new avatar image while maintaining fixed display size.
        """
        file_path = filedialog.askopenfilename(
            title="Select Avatar Image",
            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.gif")]
        )
        if not file_path:
            return

        try:
            from PIL import Image, ImageTk

            # fixed display box dimensions
            target_width = 512
            target_height = 512

            # open the selected image
            img = Image.open(file_path)

            # calculate proportional resize to fit within the target box
            img.thumbnail((target_width, target_height), Image.LANCZOS)

            # create a new blank image (with transparent or black background) that matches the fixed display box size
            from PIL import ImageOps
            fixed_img = Image.new("RGBA", (target_width, target_height), (0, 0, 0, 0))
            img_x = (target_width - img.width) // 2
            img_y = (target_height - img.height) // 2
            fixed_img.paste(img, (img_x, img_y))

            # convert to Tkinter-compatible image 
            self.image = ImageTk.PhotoImage(fixed_img)

            # update the existing label image
            self.image_label.config(image=self.image, width=target_width, height=target_height)

            # confirmation popup
            self.custom_message_popup("Avatar Changed", "AI avatar updated successfully.")
        except Exception as e:
            self.custom_message_popup("Error", f"Failed to change avatar: {e}", msg_type="error")
        
    def edit_prompt(self):
        """
        Open a popup to view or edit the current system prompt.
        """
        current_prompt = self.service.get_prompt() or "You are a friendly learning advisor who motivates students kindly."

        # use the enhanced custom_input_popup
        new_prompt = self.custom_input_popup(
            title="Edit Prompt",
            prompt=current_prompt,
            ok_text="Save",
            show_cancel=True,
            multiline=True
        )

        if new_prompt:
            self.service.set_prompt(new_prompt)
            self.custom_message_popup("Prompt Saved", "System prompt updated successfully.")

    def save_prompt(self):
        """
        Force-save current prompt text.
        """
        text = self.service.get_prompt()
        self.service.set_prompt(text)
        self.custom_message_popup("Prompt", "Prompt saved to prompt.txt.")

    def load_prompt(self):
        """
        Reload prompt from file.
        """
        text = self.service.get_prompt()
        self.custom_message_popup("Prompt Loaded", text or "No saved prompt found.")

    def refresh_session(self):
        """
        Reset LLM context and clear on-screen conversation.
        """
        msg = self.service.reset_context()
        self.ai_output_box.config(state="normal")
        self.ai_output_box.delete("1.0", tk.END)
        self.ai_output_box.insert(tk.END, "Session reset. Ready for a new conversation.\n")
        self.ai_output_box.config(state="disabled")
        self.custom_message_popup("Session", msg)

    def export_csv(self):
        """
        Export all entries (history) to a CSV file.
        Columns: EntryType, Timestamp, Text, Mood, Status
        - GoalLog adds Status
        - ReflectionLog adds Mood
        """
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv", filetypes=[("CSV files", "*.csv")]
        )
        if not file_path:
            return

        history = self.service.snapshot().entries

        from domain import GoalLog, ReflectionLog

        with open(file_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            # write header row
            writer.writerow(["EntryType", "Timestamp", "Text", "Mood", "Status"])

            # write one row per log entry
            for etype, records in history.items():
                for rec in records:
                    mood = rec.mood if hasattr(rec, "mood") else ""
                    status = ""

                    # handle derived class specifics
                    if isinstance(rec, GoalLog):
                        status = rec.status
                    elif isinstance(rec, ReflectionLog):
                        mood = rec.mood  # reflectionLog should always carry mood

                    writer.writerow([
                        etype.value,
                        rec.timestamp,
                        rec.text,
                        mood,
                        status
                    ])
        self.custom_message_popup("Exported", f"Entries exported to {file_path}", msg_type="info")

            
