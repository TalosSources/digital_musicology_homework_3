

def read_annotations(annotations_path):
    beats = []
    with open(annotations_path, 'r') as file:
        for line in file:
            columns = line.strip().split('\t')
            beats.append(float(columns[0]))
    return beats

"""
Reads a midi file with pretty midi
Param:
    piece_path (str | Path) : path of the midi file
Returns:
    a pretty_midi object
"""
def read_performed_midi(midi_path):
    ...