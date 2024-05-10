

def read_annotations(piece_path):
    beats = []
    with open(piece_path, 'r') as file:
        for line in file:
            columns = line.strip().split('\t')
            beats.append(float(columns[0]))
    return beats

def read_performed_midi(piece_path):
    ...