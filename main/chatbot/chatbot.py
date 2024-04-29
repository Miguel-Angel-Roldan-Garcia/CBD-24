import tkinter as tk
from tkinter import scrolledtext, END

from main.data.data_mining import classify

class MovieChatbot:
    def __init__(self, root, tree_version):
        self.root = root
        self.root.title("MovieGenie")
        self.root.geometry("600x600")
        self.tree_version = tree_version

        self.welcome_label = tk.Label(self.root, text="Welcome to MovieGenie", justify="center", fg="blue", font=("Times New Roman", 24))
        self.welcome_label.grid(row=0, column=0, columnspan=2, padx=10, pady=10, sticky="ew")
        
        # Create chat display
        self.chat_display = tk.Text(root, width=70, height=20, state="disabled")
        self.chat_display.grid(row=1, column=0, columnspan=2, padx=10, pady=10, sticky="ew")

        
        # Create user input entry
        self.user_input = tk.Entry(root, width=50)
        self.user_input.grid(row=2, column=0, padx=10, pady=10, sticky="ew")

        # Create send button
        self.send_button = tk.Button(root, text="Send", command=self.send_message)
        self.send_button.grid(row=2, column=1, padx=10, pady=10, sticky="ew")
        self.root.bind('<Return>', self.send_message_event)

        # Configure grid weights to allow resizing
        for i in range(3):
            self.root.grid_rowconfigure(i, weight=1)
        for j in range(2):
            self.root.grid_columnconfigure(j, weight=1)

    def classify_genre(self, description):
        return classify(description, self.tree_version)

    def send_message(self):
        user_message = self.user_input.get().strip()  # Remove leading and trailing whitespace
        if user_message:
            self.chat_display.configure(state='normal')
            self.chat_display.insert(tk.END, "You: " + user_message + "\n")
            self.user_input.delete(0, tk.END)
            if user_message.lower() == 'bye':
                self.chat_display.insert(tk.END, "Chatbot: Goodbye!\n")
            else:
                genre = self.classify_genre(user_message)
                self.chat_display.insert(tk.END, f"Chatbot: This movie could be categorized as '{genre}'.\n")
            self.chat_display.configure(state='disabled')
            self.chat_display.see(tk.END)  # Scroll to the bottom of the chat display
        else:
            # Display a message to the user indicating that the message is empty
            self.chat_display.insert(tk.END, "[Message cannot be empty]\n")
            self.chat_display.see(tk.END)  # Scroll to the bottom of the chat display

    def send_message_event(self, event):
        self.send_message()
        
    def mainloop(self):
        self.root.mainloop()

def run(tree_version = 0):
    root = tk.Tk()

    chatbot = MovieChatbot(root, tree_version)
    chatbot.mainloop()