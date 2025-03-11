labels = [
    # Music
    "Acoustic_guitar",
    "Bass_guitar",
    "Bowed_string_instrument",
    "Crash_cymbal",
    "Electric_guitar",
    "Gong",
    "Harp",
    "Organ",
    "Piano",
    "Rattle_(instrument)",
    "Scratching_(performance_technique)",
    "Snare_drum",
    "Trumpet",
    "Wind_chime",
    "Wind_instrument_and_woodwind_instrument",
    "Livestock_and_farm_animals_and_working_animals",
    # Sounds of things
    "Boom",
    "Camera",
    "Coin_(dropping)",
    "Computer_keyboard",
    "Crack",
    "Dishes_and_pots_and_pans",
    "Drawer_open_or_close",
    "Drill",
    "Gunshot_and_gunfire",
    "Hammer",
    "Keys_jangling",
    "Knock",
    "Microwave_oven",
    "Printer",
    "Sawing",
    "Scissors",
    "Skateboard",
    "Slam",
    "Splash_and_splatter",
    "Squeak",
    "Tap",
    "Thump_and_thud",
    "Toilet_flush",
    "Train",
    "Water_tap_and_faucet",
    "Whoosh_and_swoosh_and_swish",
    "Writing",
    "Zipper_(clothing)",
    # Natural sounds
    "Crackle",
    "Stream",
    "Waves_and_surf",
    "Wind",
    # Human sounds
    "Burping_and_eructation",
    "Chewing_and_mastication",
    "Child_speech_and_kid_speaking",
    "Clapping",
    "Cough",
    "Crying_and_sobbing",
    "Fart",
    "Female_singing",
    "Female_speech_and_woman_speaking",
    "Finger_snapping",
    "Giggle",
    "Male_speech_and_man_speaking",
    "Run",
    "Screaming",
    "Walk_and_footsteps",
    # Animal
    "Bark",
    "Cricket",
    "Livestock,_farm_animals,_working_animals",
    "Meow",
    "Rattle",
    # Source-ambiguous sounds
    "Crumpling_and_crinkling",
    "Crushing",
    "Tearing",
]


def label_to_number(label):
    """Encode text label as number between 0 and label_count() - 1"""
    return label.index(label)


def number_to_label(number):
    """Decode text label from number to string"""
    return labels[number]


def label_count():
    """Get number of labels"""
    return len(labels)
