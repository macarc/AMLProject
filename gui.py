from tkinter import *
from tkinter import ttk, filedialog
import glob
import torch
import pygame

import convnet
from helpers import get_torch_backend, load_model
from labels import label_count, number_to_label
from load_datasets import load_files


class AudioPlayer:
    def __init__(self):
        """Set up audio playback"""
        pygame.mixer.init()
        self.current_sound = None

    def play(self, filename):
        """Play WAV file, first stopping any currently playing audio"""
        self.stop()
        self.current_sound = pygame.mixer.Sound(filename)
        self.current_sound.play()

    def stop(self):
        """Stop any currently-playing audio file"""
        if self.current_sound:
            self.current_sound.stop()


class AudioFileList:
    def __init__(self, parent):
        """Create AudioFileList widget"""
        self.file_list = []
        self.labels = []

        self.tree = ttk.Treeview(parent, columns=2)
        self.tree.heading("#0", text="Effect type", anchor="center")
        self.tree.column("#0", stretch=False)
        self.tree.heading("#1", text="File name", anchor="center")
        self.tree.grid(column=2, row=1, columnspan=2, sticky="news")

    def widget(self):
        """Get the underlying root widget (useful for positioning)"""
        return self.tree

    def set_audio_files(self, file_list, labels):
        """Set the list of audio files that the user has opened."""
        self.file_list = file_list
        self.labels = labels
        self.view()

    def view(self, selected="all"):
        """Redraw the list of widgets. `selected` is either 'all' or a list of labels (as numbers)"""
        self.tree.delete(*self.tree.get_children())
        for i in range(len(self.file_list)):
            if selected == "all" or self.labels[i] in selected:
                self.tree.insert(
                    "",
                    "end",
                    text=self._get_label_name(i),
                    values=(self._get_audio_name(i),),
                    iid=self._get_audio_name(i),
                )

    def selection(self):
        """Get currently selected audio files - if only one audio file is viewed, then that is treated as selected"""
        if len(self.tree.get_children()) == 1:
            return self.tree.get_children()
        else:
            return self.tree.selection()

    def _get_label_name(self, label_index):
        return number_to_label(self.labels[label_index]).replace("_", " ")

    def _get_audio_name(self, index):
        return self.file_list[index]


class LabelList:
    def __init__(self, parent, on_select):
        """Create LabelList widget"""
        self.on_select = on_select

        # Set up list
        self.tree = ttk.Treeview(parent)
        self.tree.column("#0", width=100)
        self.tree.heading("#0", text="Filter by...", anchor="center")
        self.tree.bind("<<TreeviewSelect>>", self.select)

        # Add 'all' label
        self.tree.insert("", "end", text="All", iid="all")

        # Add each label to the list
        for i in range(label_count()):
            self.tree.insert(
                "", "end", text=number_to_label(i).replace("_", " "), iid=i
            )

    def widget(self):
        """Get the underlying root widget (useful for positioning)"""
        return self.tree

    def select_all(self):
        """Select 'all' - to show all audio files"""
        self.tree.focus_set()
        self.tree.selection_set("all")
        self.tree.focus("all")

    def select(self, e):
        """Callback for when selection has changed - call self.on_select with the new selected labels"""
        if "all" in self.tree.selection():
            self.on_select("all")
        else:
            self.on_select([int(l) for l in self.tree.selection()])


class App:
    def __init__(self, nnet, extract_features):
        """Set up the GUI application"""
        self.nnet = nnet
        self.extract_features = extract_features

        self.audio_player = AudioPlayer()

        # Set up window
        self.root = Tk()
        self.root.geometry("800x600")
        self.root.title("Audio file explorer")
        self.root.grid()
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)

        # Set up frame containing all widgets
        frm = ttk.Frame(self.root)
        frm.grid(stick="news")

        # Set up buttons
        ttk.Button(frm, text="Select file(s)", command=self.open_files).grid(
            column=0, row=0
        )
        ttk.Button(frm, text="Select directory", command=self.open_dir).grid(
            column=1, row=0
        )
        ttk.Button(frm, text="Play", command=self.play).grid(column=2, row=0)
        ttk.Button(frm, text="Stop", command=self.stop).grid(column=3, row=0)

        # Set up lists
        self.audio_file_list = AudioFileList(frm)
        self.label_list = LabelList(frm, self.audio_file_list.view)

        self.label_list.widget().grid(column=0, row=1, columnspan=2, stick="news")
        self.audio_file_list.widget().grid(column=2, row=1, columnspan=2, sticky="news")

        # Configure shrink/resize behaviour
        frm.columnconfigure((0, 1), weight=1)
        frm.columnconfigure((2, 3), weight=2)
        frm.rowconfigure(1, weight=1)

    def run(self):
        """Run the GUI application (returns only when the app has been closed)"""
        self.root.mainloop()

    def play(self):
        """Get the selected audio file from the AudioFileList and play using AudioPlayer"""
        selected_audio_files = self.audio_file_list.selection()
        if len(selected_audio_files) > 0:
            self.audio_player.play(selected_audio_files[0])

    def stop(self):
        """Stop playing the current audio file"""
        self.audio_player.stop()

    def open_files(self):
        """Prompt the user to open audio files, and then update the AudioFileList accordingly"""
        filenames = filedialog.askopenfilenames()
        if len(filenames) > 0:
            self._load(filenames)

    def open_dir(self):
        """Prompt the user to open a directory, and then update the AudioFileList accordingly"""
        d = filedialog.askdirectory()
        if d:
            filenames = [
                f"{d}/{f}" for f in glob.glob("**/*.wav", root_dir=d, recursive=True)
            ]
            self._load(filenames)

    def _load(self, filenames):
        """Load the features from filenames, get the predictions, and update the AudioFileList"""
        # Get features and predicted output
        features = load_files(filenames, backend_dev, self.extract_features)
        output = self.nnet(features)

        # Fix for if only one audio file is selected
        if len(output.shape) == 1:
            output = output.unsqueeze(1)

        # Get the predicted labels
        labels = list(torch.argmax(output, dim=1))

        # Show the files
        self.audio_file_list.set_audio_files(filenames, labels)

        # Select 'all' category so that all labels are shown
        self.label_list.select_all()


# Load the ConvNet model from the file
backend_dev = get_torch_backend()
model_filename = "models/conv.pt"

nnet = convnet.ConvNet([128, 64, 64, 64, 70], label_count(), 5)
optimiser = torch.optim.Adam(nnet.parameters())
nnet.to(backend_dev)

if not load_model(model_filename, nnet, optimiser):
    raise Exception("Couldn't load model!")

# Create the GUI and run it
App(nnet, convnet.extract_features).run()
