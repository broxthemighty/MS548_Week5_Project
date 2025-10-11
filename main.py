"""
main.py
Author: Matt Lindborg
Course: MS548 - Advanced Programming Concepts and AI
Assignment: Week 5
Date: 10/11/2025

Purpose:
This is the entry point for the Learnflow Base application.
It wires together the user interface (ui.py) with the service layer (service.py).
The structure follows best practices:
    - Keep main.py minimal (only startup logic).
    - Delegate business logic to service.py.
    - Delegate GUI rendering to ui.py.
"""

# --- Imports ---
import tkinter as tk                  # Tkinter for GUI window creation
from service import LearnflowService  # service layer for business logic
from ui import App                    # GUI class

def main():
    """
    Application entry function.
    Creates the root Tkinter window, service instance, and App GUI.
    Starts the event loop to keep the program running until exit.
    """
    # create root Tkinter window
    root_window = tk.Tk()

    # create service instance
    service = LearnflowService()

    # build GUI, passing in root window and service args
    app = App(root_window, service)

    # enter Tkinter event loop 
    root_window.mainloop()

# python standard entry-point
if __name__ == "__main__":
    main()
